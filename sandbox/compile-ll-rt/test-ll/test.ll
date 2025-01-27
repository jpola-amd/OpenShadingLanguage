; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fabs.f32(float) #0

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.sqrt.f32(float) #0

; Function Attrs: alwaysinline mustprogress nofree norecurse nosync nounwind willreturn memory(none)
declare dso_local noundef nofpclass(nan inf) float @osl_fabs_ff(float noundef nofpclass(nan inf)) local_unnamed_addr #1

; Function Attrs: alwaysinline mustprogress nofree norecurse nosync nounwind willreturn memory(none)
declare dso_local nofpclass(nan inf) float @osl_safe_div_fff(float noundef nofpclass(nan inf), float noundef nofpclass(nan inf)) local_unnamed_addr #1

; Function Attrs: alwaysinline mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: read)
declare dso_local noundef nofpclass(nan inf) float @osl_filterwidth_fdf(ptr nocapture noundef readonly) local_unnamed_addr #2

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #3

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, inaccessiblemem: none)
define void @__direct_callable__osl_layer_group_unnamed_group_1_name_test_0(ptr nocapture readonly %0, ptr nocapture %1, ptr nocapture readnone %2, ptr %3, i32 %4, ptr nocapture readnone %5) local_unnamed_addr #4 {
bb___direct_callable__osl_layer_group_unnamed_group_1_name_test_0_11:
  %6 = getelementptr i8, ptr %1, i64 4
  store float 0.000000e+00, ptr %6, align 4
  %7 = getelementptr i8, ptr %1, i64 8
  store float 0.000000e+00, ptr %7, align 4
  %8 = getelementptr i8, ptr %1, i64 12
  store float 0.000000e+00, ptr %8, align 4
  %9 = getelementptr i8, ptr %0, i64 108
  %10 = load float, ptr %9, align 4
  %11 = fadd float %10, -5.000000e-01
  %12 = getelementptr i8, ptr %0, i64 112
  %13 = load float, ptr %12, align 4
  %14 = getelementptr i8, ptr %0, i64 116
  %15 = load float, ptr %14, align 4
  %16 = getelementptr i8, ptr %0, i64 120
  %17 = load float, ptr %16, align 4
  %18 = fmul float %17, 2.500000e-01
  %19 = getelementptr i8, ptr %0, i64 124
  %20 = load float, ptr %19, align 4
  %21 = fmul float %18, 0.000000e+00
  %22 = fsub float %20, %21
  %23 = fmul float %22, 2.500000e-01
  %24 = getelementptr i8, ptr %0, i64 128
  %25 = load float, ptr %24, align 4
  %26 = fsub float %25, %21
  %27 = fmul float %26, 2.500000e-01
  %28 = fsub float %11, %18
  %29 = fsub float %13, %23
  %30 = fsub float %15, %27
  %31 = fmul fast float %29, %29
  %32 = fmul fast float %30, %30
  %33 = fadd fast float %32, %31
  %34 = tail call fast noundef float @llvm.sqrt.f32(float %33)
  %35 = fmul float %34, 5.000000e-01
  %36 = fsub float 0.000000e+00, %35
  %37 = fcmp ule float %28, %36
  br i1 %37, label %osl_layer_group_unnamed_group_1_name_test_0.exit, label %bb_else_5.i

bb_else_5.i:                                      ; preds = %bb___direct_callable__osl_layer_group_unnamed_group_1_name_test_0_11
  %38 = fcmp uge float %28, %35
  br i1 %38, label %osl_layer_group_unnamed_group_1_name_test_0.exit, label %bb_else_8.i

bb_else_8.i:                                      ; preds = %bb_else_5.i
  %39 = fsub float %28, %36
  %40 = fdiv fast float %39, %34
  br label %osl_layer_group_unnamed_group_1_name_test_0.exit

osl_layer_group_unnamed_group_1_name_test_0.exit: ; preds = %bb___direct_callable__osl_layer_group_unnamed_group_1_name_test_0_11, %bb_else_5.i, %bb_else_8.i
  %.0.i = phi float [ %40, %bb_else_8.i ], [ 0.000000e+00, %bb___direct_callable__osl_layer_group_unnamed_group_1_name_test_0_11 ], [ 1.000000e+00, %bb_else_5.i ]
  %41 = fsub float 1.000000e+00, %.0.i
  %42 = fmul float %41, 0x3FECCCCCC0000000
  %43 = fmul float %.0.i, 0x3FA99999A0000000
  %44 = fadd float %43, %42
  store float %44, ptr %6, align 4
  store float %44, ptr %7, align 4
  %45 = fmul float %.0.i, 0x3FECCCCCC0000000
  %46 = fadd float %45, %42
  store float %46, ptr %8, align 4
  %47 = sext i32 %4 to i64
  %48 = mul nsw i64 %47, 12
  %49 = ptrtoint ptr %3 to i64
  %50 = add i64 %48, %49
  %51 = inttoptr i64 %50 to ptr
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(12) %51, ptr noundef nonnull align 1 dereferenceable(12) %6, i64 12, i1 false)
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none)
define void @__direct_callable__osl_init_group_unnamed_group_1(ptr nocapture readnone %0, ptr nocapture readnone %1, ptr nocapture readnone %2, ptr nocapture readnone %3, i32 %4, ptr nocapture readnone %5) local_unnamed_addr #5 {
bb___direct_callable__osl_init_group_unnamed_group_1_12:
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, inaccessiblemem: none)
define void @__direct_callable__fused_unnamed_group_1_name_test_0(ptr nocapture readonly %0, ptr nocapture %1, ptr nocapture readnone %2, ptr %3, i32 %4, ptr nocapture readnone %5) local_unnamed_addr #4 {
bb___direct_callable__fused_unnamed_group_1_name_test_0_13:
  %6 = getelementptr i8, ptr %1, i64 4
  store float 0.000000e+00, ptr %6, align 4
  %7 = getelementptr i8, ptr %1, i64 8
  store float 0.000000e+00, ptr %7, align 4
  %8 = getelementptr i8, ptr %1, i64 12
  store float 0.000000e+00, ptr %8, align 4
  %9 = getelementptr i8, ptr %0, i64 108
  %10 = load float, ptr %9, align 4
  %11 = fadd float %10, -5.000000e-01
  %12 = getelementptr i8, ptr %0, i64 112
  %13 = load float, ptr %12, align 4
  %14 = getelementptr i8, ptr %0, i64 116
  %15 = load float, ptr %14, align 4
  %16 = getelementptr i8, ptr %0, i64 120
  %17 = load float, ptr %16, align 4
  %18 = fmul float %17, 2.500000e-01
  %19 = getelementptr i8, ptr %0, i64 124
  %20 = load float, ptr %19, align 4
  %21 = fmul float %18, 0.000000e+00
  %22 = fsub float %20, %21
  %23 = fmul float %22, 2.500000e-01
  %24 = getelementptr i8, ptr %0, i64 128
  %25 = load float, ptr %24, align 4
  %26 = fsub float %25, %21
  %27 = fmul float %26, 2.500000e-01
  %28 = fsub float %11, %18
  %29 = fsub float %13, %23
  %30 = fsub float %15, %27
  %31 = fmul fast float %29, %29
  %32 = fmul fast float %30, %30
  %33 = fadd fast float %32, %31
  %34 = tail call fast noundef float @llvm.sqrt.f32(float %33)
  %35 = fmul float %34, 5.000000e-01
  %36 = fsub float 0.000000e+00, %35
  %37 = fcmp ule float %28, %36
  br i1 %37, label %osl_layer_group_unnamed_group_1_name_test_0.exit, label %bb_else_5.i

bb_else_5.i:                                      ; preds = %bb___direct_callable__fused_unnamed_group_1_name_test_0_13
  %38 = fcmp uge float %28, %35
  br i1 %38, label %osl_layer_group_unnamed_group_1_name_test_0.exit, label %bb_else_8.i

bb_else_8.i:                                      ; preds = %bb_else_5.i
  %39 = fsub float %28, %36
  %40 = fdiv fast float %39, %34
  br label %osl_layer_group_unnamed_group_1_name_test_0.exit

osl_layer_group_unnamed_group_1_name_test_0.exit: ; preds = %bb___direct_callable__fused_unnamed_group_1_name_test_0_13, %bb_else_5.i, %bb_else_8.i
  %.0.i = phi float [ %40, %bb_else_8.i ], [ 0.000000e+00, %bb___direct_callable__fused_unnamed_group_1_name_test_0_13 ], [ 1.000000e+00, %bb_else_5.i ]
  %41 = fsub float 1.000000e+00, %.0.i
  %42 = fmul float %41, 0x3FECCCCCC0000000
  %43 = fmul float %.0.i, 0x3FA99999A0000000
  %44 = fadd float %43, %42
  store float %44, ptr %6, align 4
  store float %44, ptr %7, align 4
  %45 = fmul float %.0.i, 0x3FECCCCCC0000000
  %46 = fadd float %45, %42
  store float %46, ptr %8, align 4
  %47 = sext i32 %4 to i64
  %48 = mul nsw i64 %47, 12
  %49 = ptrtoint ptr %3 to i64
  %50 = add i64 %48, %49
  %51 = inttoptr i64 %50 to ptr
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(12) %51, ptr noundef nonnull align 1 dereferenceable(12) %6, i64 12, i1 false)
  ret void
}

